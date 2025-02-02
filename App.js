import React, { useState } from "react";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import "./App.css";

export default function Chatbot() {
  const [mensaje, setMensaje] = useState("");
  const [conversacion, setConversacion] = useState([]);
  const [escribiendo, setEscribiendo] = useState(false); 

  const enviarMensaje = async () => {
    if (!mensaje.trim()) return;

    //Agregar mensaje del usuario a la conversacion
    setConversacion((prev) => [...prev, { user: true, text: mensaje }]);
    setEscribiendo(true); // Activar Escribiendo

    try {
      const response = await fetch("https://servidor-backend.ngrok.app/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta: mensaje }),
      });

      const data = await response.json();
      setEscribiendo(false); // Desactivar "Escribiendo..."
      setConversacion((prev) => [...prev, { user: false, text: data.respuesta }]);
    } catch (error) {
      console.error("Error al conectar con el backend:", error);
      setEscribiendo(false);
      setConversacion((prev) => [
        ...prev,
        { user: false, text: "Hubo un error al obtener la respuesta." },
      ]);
    }

    setMensaje("");
  };

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Chatbot</h1>
      <div className="border p-4 rounded-md">
        <div className="h-64 overflow-auto border-b pb-4">
          {conversacion.map((msg, index) => (
            <p key={index} className={msg.user ? "text-blue-600" : "text-gray-800"}>
              {msg.text}
            </p>
          ))}
          {escribiendo && <p className="text-gray-500 italic">Escribiendo...</p>}
        </div>
        <div className="mt-4 flex">
          <Input
            value={mensaje}
            onChange={(e) => setMensaje(e.target.value)}
            placeholder="Escribe tu pregunta..."
          />
          <Button onClick={enviarMensaje} className="ml-2">
            Enviar
          </Button>
        </div>
      </div>
    </div>
  );
}
